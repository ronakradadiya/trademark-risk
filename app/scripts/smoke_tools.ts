import { lookupApplicantHistory } from '../src/tools/lookup_applicant_history.js';
import { checkUsptoMarks } from '../src/tools/check_uspto_marks.js';

async function run(): Promise<void> {
  const header = (s: string): void => console.log(`\n=== ${s} ===`);

  header('lookup_applicant_history: Apple Inc.');
  const apple = await lookupApplicantHistory({ applicant_name: 'Apple Inc.' });
  console.log(apple);

  header('lookup_applicant_history: Nike Inc.');
  const nike = await lookupApplicantHistory({ applicant_name: 'Nike Inc.' });
  console.log(nike);

  header('lookup_applicant_history: Three Hearts Products, LLC (known troll)');
  const troll = await lookupApplicantHistory({ applicant_name: 'Three Hearts Products, LLC' });
  console.log(troll);

  header('lookup_applicant_history: I420 LLC (known troll)');
  const i420 = await lookupApplicantHistory({ applicant_name: 'I420 LLC' });
  console.log(i420);

  header('lookup_applicant_history: Nonsense Holdings ZZZ (unknown)');
  const unknown = await lookupApplicantHistory({ applicant_name: 'Nonsense Holdings ZZZ' });
  console.log(unknown);

  header('check_uspto_marks: NOVAPAY');
  const novapay = await checkUsptoMarks({ brand_name: 'NovaPay' });
  console.log(novapay);

  header('check_uspto_marks: NIKE class 25');
  const nikeMarks = await checkUsptoMarks({ brand_name: 'Nike', class_code: 25 });
  if (nikeMarks.ok) {
    console.log('total:', nikeMarks.data.total);
    console.log('sample:', nikeMarks.data.results.slice(0, 3));
  }

  header('check_uspto_marks: xyzzy99999abc (no match)');
  const nothing = await checkUsptoMarks({ brand_name: 'xyzzy99999abc' });
  console.log(nothing);
}

run().catch((e) => {
  console.error(e);
  process.exit(1);
});
